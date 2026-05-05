use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Attribute, Expr, Ident, Token, Visibility};

struct DefinePipelineInput {
    attrs: Vec<Attribute>,
    vis: Visibility,
    name: Ident,
    config: Expr,
}

impl Parse for DefinePipelineInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        input.parse::<Token![struct]>()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        // config = <expr>;
        let ident: Ident = input.parse()?;
        assert_eq!(ident, "config", "Expected 'config'");
        input.parse::<Token![=]>()?;
        let config: Expr = input.parse()?;
        input.parse::<Token![;]>()?;

        Ok(DefinePipelineInput {
            attrs,
            vis,
            name,
            config,
        })
    }
}

pub fn define_graphics_pipeline_impl(input: TokenStream) -> TokenStream {
    let DefinePipelineInput {
        attrs,
        vis,
        name,
        config,
    } = syn::parse_macro_input!(input as DefinePipelineInput);

    let expanded = quote! {
        #(#attrs)*
        #vis struct #name;

        impl crate::vulkan::resource_config::GraphicsPipelineMarker for #name {
            const CONFIG: crate::vulkan::pipeline_config::GraphicsPipelineConfig = #config;

            fn id() -> &'static std::sync::atomic::AtomicUsize {
                // SAFETY: AtomicUsize and usize have identical memory layout.
                static ID: std::sync::atomic::AtomicUsize =
                    unsafe { std::mem::transmute(usize::MAX) };
                &ID
            }
        }
    };

    expanded.into()
}

pub fn define_compute_pipeline_impl(input: TokenStream) -> TokenStream {
    let DefinePipelineInput {
        attrs,
        vis,
        name,
        config,
    } = syn::parse_macro_input!(input as DefinePipelineInput);

    let expanded = quote! {
        #(#attrs)*
        #vis struct #name;

        impl crate::vulkan::resource_config::ComputePipelineMarker for #name {
            const CONFIG: crate::vulkan::pipeline_config::ComputePipelineConfig = #config;

            fn id() -> &'static std::sync::atomic::AtomicUsize {
                // SAFETY: AtomicUsize and usize have identical memory layout.
                static ID: std::sync::atomic::AtomicUsize =
                    unsafe { std::mem::transmute(usize::MAX) };
                &ID
            }
        }
    };

    expanded.into()
}
